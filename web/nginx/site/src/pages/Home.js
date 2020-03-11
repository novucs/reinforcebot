import React, {Component} from 'react'
import {Button, Container, Grid, Header, Icon, Image, Menu, Segment, Visibility,} from 'semantic-ui-react'
import whiteLogo from "../white-logo.svg";
import problemImg from "../problem.png";
import solutionImg from "../solution.png";
import Footer from "../Footer";

const HomepageHeading = ({}) => (
  <Container text>
    <img
      style={{
        marginBottom: '3em',
        marginTop: '12em'
      }}
      src={whiteLogo}
      alt="logo"
      className="image"
    />
    <Button primary size='huge'>
      Get Started
      <Icon name='right arrow'/>
    </Button>
  </Container>
);

export default class HomepageLayout extends Component {
  constructor(props) {
    super(props);
    this.state = {fixed: false};
  }

  hideFixedMenu = () => this.setState({fixed: false});
  showFixedMenu = () => this.setState({fixed: true});

  render() {
    return (
      <div>
        <Visibility
          once={false}
          onBottomPassed={this.showFixedMenu}
          onBottomPassedReverse={this.hideFixedMenu}
        >
          <Segment
            inverted
            textAlign='center'
            style={{minHeight: 700, padding: '1em 0em'}}
            vertical
          >
            <Menu
              fixed={this.state.fixed ? 'top' : null}
              inverted={!this.state.fixed}
              pointing={!this.state.fixed}
              secondary={!this.state.fixed}
              size='large'
            >
              <Container>
                <Menu.Item as='a' active>
                  Home
                </Menu.Item>
                <Menu.Item position='right'>
                  <Button as='a' href='/signin' inverted={!this.state.fixed}>
                    Sign in
                  </Button>
                  <Button as='a' href='/signup' inverted={!this.state.fixed} primary={this.state.fixed}
                          style={{marginLeft: '0.5em'}}>
                    Sign Up
                  </Button>
                </Menu.Item>
              </Container>
            </Menu>
            <HomepageHeading/>
          </Segment>
        </Visibility>

        <Segment style={{padding: '8em 0em'}} vertical>
          <Grid container stackable verticalAlign='middle'>
            <Grid.Row>
              <Grid.Column width={8}>
                <Header as='h3' style={{fontSize: '2em'}}>
                  Ever been tired of repetitive tasks?
                </Header>
                <p style={{fontSize: '1.33em'}}>
                  Tasks on your computer may at time to time become repetitive.

                  The only two solutions around include creating macros, and developing an agent to automate the
                  workflow.

                  However, macros are brittle, and hiring a developer to create an agent can be expensive.
                </p>
              </Grid.Column>
              <Grid.Column floated='right' width={6}>
                <Image bordered rounded size='medium' src={problemImg}/>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Segment>

        <Segment style={{padding: '8em 0em'}} vertical>
          <Grid container stackable verticalAlign='middle'>
            <Grid.Row>
              <Grid.Column floated='left' width={6}>
                <Image bordered rounded size='medium' src={solutionImg}/>
              </Grid.Column>
              <Grid.Column width={8}>
                <Header as='h3' style={{fontSize: '2em'}}>
                  ReinforceBot automates your tasks
                </Header>
                <p style={{fontSize: '1.33em'}}>
                  ReinforceBot enables you to create custom agents with no prior
                  development experience required. These agents will be happy to
                  automate your task for you, and are even capable of playing games!
                </p>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Segment>
        <Footer/>
      </div>
    );
  }
}
