import React, {Component} from "react";
import {Button, Container, Dropdown, Icon, Menu} from "semantic-ui-react";
import {BASE_URL, getJWT, hasJWT} from "./Util";

export default class TopMenu extends Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  signOut() {
    window.localStorage.setItem('jwtAccess', null);
    window.localStorage.setItem('jwtRefresh', null);
    window.location = '/signin';
  }

  componentDidMount() {
    if (hasJWT()) {
      fetch(BASE_URL + '/api/auth/users/me/', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Authorization': 'JWT ' + getJWT(),
        },
      }).then(response => {
        if (response.status < 200 || response.status >= 300) {
          this.signOut();
          return;
        }

        this.setState({errors: []});
        response.json().then(body => {
          this.setState({
            username: body['username'],
            firstName: body['first_name'],
            lastName: body['last_name'],
          });
        });
      });
    }
  }

  render() {


    return (
      <Menu fixed='top' size='large' inverted>
        <Container>
          <Menu.Item as='a' href='/'>
            Home
          </Menu.Item>
          <Menu.Item as='a' href='/dashboard'>
            Dashboard
          </Menu.Item>
          <Menu.Item position='right'>
            <div hidden={hasJWT()}>
              <Button as='a' href='/signin'>
                Sign in
              </Button>
              <Button as='a' href='/signup' primary style={{marginLeft: '0.5em'}}>
                Sign Up
              </Button>
            </div>
            <div hidden={!hasJWT()}>
              <Dropdown
                className="simple"
                trigger={(
                  <span>
                    <Icon name='user'/> Hello, {this.state.username}
                  </span>
                )}
                options={[
                  {
                    key: 'user',
                    text: (
                      <span>
                        Signed in as <strong>{this.state.firstName} {this.state.lastName}</strong>
                      </span>
                    ),
                    disabled: true,
                  },
                  {key: 'sign-out', text: 'Sign Out', icon: 'sign out', onClick: this.signOut},
                ]}
              />
            </div>
          </Menu.Item>
        </Container>
      </Menu>
    )
  }
}
