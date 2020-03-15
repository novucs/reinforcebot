import React, {Component} from "react";
import {Button, Container, Dropdown, Icon, Menu} from "semantic-ui-react";
import {hasJWT, signOut} from "../Util";

class UserDropdown extends Component {
  // props:
  // me: User

  render() {
    if (this.props.me === undefined) return null;

    return (
      <div hidden={!hasJWT()}>
        <Dropdown
          className="simple"
          trigger={(
            <span>
              <Icon name='user'/> Hello, {this.props.me.username}
            </span>
          )}
          options={[
            {
              key: 'user',
              text: (
                <span>
                  Signed in as <strong>{this.props.me.first_name} {this.props.me.last_name}</strong>
                </span>
              ),
              disabled: true,
            },
            {key: 'profile', text: 'Profile', icon: 'user', onClick: () => {window.location = '/profile'}},
            {key: 'sign-out', text: 'Sign Out', icon: 'sign out', onClick: signOut},
          ]}
        />
      </div>
    );
  }
}

export default class TopMenu extends Component {
  // props:
  // me: User

  render() {
    return (
      <Menu fixed='top' size='large' inverted>
        <Container>
          <Menu.Item as='a' href='/'>
            Home
          </Menu.Item>
          <Menu.Item as='a' href='/agents'>
            Agents
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
            <UserDropdown me={this.props.me}/>
          </Menu.Item>
        </Container>
      </Menu>
    )
  }
}
